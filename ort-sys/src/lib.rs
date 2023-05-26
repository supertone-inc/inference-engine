#![allow(non_snake_case)]
#![allow(clippy::missing_safety_doc)]

include!("bindings.rs");

impl<E: From<ResultCode>> From<ResultCode> for Result<(), E> {
    fn from(code: ResultCode) -> Self {
        match code {
            ResultCode::Ok => Ok(()),
            _ => Err(code.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::{c_void, CStr};
    use std::ptr::{null, null_mut};

    #[derive(Debug)]
    struct Error(String);

    impl From<ResultCode> for Error {
        fn from(code: ResultCode) -> Self {
            match code {
                ResultCode::Ok => panic!("expected error"),
                ResultCode::Error => unsafe {
                    Self(
                        CStr::from_ptr(get_last_error_message())
                            .to_string_lossy()
                            .into(),
                    )
                },
            }
        }
    }

    type Result<T> = std::result::Result<T, Error>;

    struct Engine(*mut c_void);

    impl Drop for Engine {
        fn drop(&mut self) {
            unsafe {
                Result::from(destroy_inference_engine(self.0)).unwrap();
            }
        }
    }

    #[test]
    fn with_invalid_model_data() {
        unsafe {
            match Result::from(create_inference_engine(null(), 0, null_mut())) {
                Ok(_) => panic!("expected error"),
                Err(Error(message)) => assert_eq!(message, "No graph was found in the protobuf."),
            }
        }
    }

    #[test]
    fn with_dynamic_shape_model() {
        unsafe {
            let model_data = include_bytes!("../../ort-cpp/test-models/matmul_dynamic.onnx");

            let mut engine = Engine(null_mut());
            Result::from(create_inference_engine(
                model_data.as_ptr() as _,
                model_data.len(),
                &mut engine.0,
            ))
            .unwrap();

            assert_eq!(get_input_count(engine.0), 2);
            assert_eq!(get_output_count(engine.0), 1);

            assert_eq!(get_input_shapes(engine.0), [[0, 0], [0, 0]]);
            assert_eq!(get_output_shapes(engine.0), [[0, 0]]);

            set_input_shapes(engine.0, &[&[2, 1], &[1, 2]]);
            assert_eq!(get_input_shapes(engine.0), [[2, 1], [1, 2]]);

            set_output_shapes(engine.0, &[&[2, 2]]);
            assert_eq!(get_output_shapes(engine.0), [[2, 2]]);

            let input_data = [[1., 2.], [3., 4.]];
            for i in 0..get_input_count(engine.0) {
                Result::from(set_input_data(engine.0, i, input_data[i].as_ptr())).unwrap();
                assert_eq!(get_input_data(engine.0, i), input_data[i].as_ptr() as _);
            }

            let mut output_data = [[0., 0., 0., 0.]];
            for i in 0..get_output_count(engine.0) {
                Result::from(set_output_data(engine.0, i, output_data[i].as_mut_ptr())).unwrap();
                assert_eq!(get_output_data(engine.0, i), output_data[i].as_ptr() as _);
            }

            Result::from(run(engine.0)).unwrap();
            assert_eq!(output_data, [[3., 4., 6., 8.]]);
        }
    }

    unsafe fn get_input_shapes(engine: *const c_void) -> Vec<Vec<usize>> {
        let mut input_shapes = vec![];
        for i in 0..get_input_count(engine) {
            let mut data = null();
            let mut size = 0;
            get_input_shape(engine, i, &mut data, &mut size);
            input_shapes.push(std::slice::from_raw_parts(data, size).into());
        }
        input_shapes
    }

    unsafe fn get_output_shapes(engine: *const c_void) -> Vec<Vec<usize>> {
        let mut output_shapes = vec![];
        for i in 0..get_output_count(engine) {
            let mut data = null();
            let mut size = 0;
            get_output_shape(engine, i, &mut data, &mut size);
            output_shapes.push(std::slice::from_raw_parts(data, size).into());
        }
        output_shapes
    }

    unsafe fn set_input_shapes(engine: *mut c_void, shapes: &[&[usize]]) {
        for i in 0..get_input_count(engine) {
            Result::from(set_input_shape(
                engine,
                i,
                shapes[i].as_ptr(),
                shapes[i].len(),
            ))
            .unwrap();
        }
    }

    unsafe fn set_output_shapes(engine: *mut c_void, shapes: &[&[usize]]) {
        for i in 0..get_output_count(engine) {
            Result::from(set_output_shape(
                engine,
                i,
                shapes[i].as_ptr(),
                shapes[i].len(),
            ))
            .unwrap();
        }
    }
}
